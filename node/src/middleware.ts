
/**
 * Middleware for LanceDB Connection. This allows you to enhance the behaviour of
 * LanceDB Connection
 */
export interface ConnectionMiddleware {
  /**
   * A callback that can be used to instrument the behaviour of http requests to remote
   * tables
   * @param {RemoteRes} res - Request ot the remote endpoint
   * @param {onRemoteRequestNext} next - Callback to advance the middleware chain
   * @param {MiddlewareContext} ctx - Local context for ths invocation of the middleware
   */
  onRemoteRequest(
    req: RemoteRequest,
    next: onRemoteRequestNext,
    ctx: MiddlewareContext,
  ): Promise<RemoteRes>
};

/**
   * A callback that can be used to instrument the behaviour of http requests to remote
   * tables
   */
export interface TableMiddleware {
  /**
   * A callback that can be used to instrument the behaviour of http requests to remote
   * tables
   * @param {RemoteRes} res - Request ot the remote endpoint
   * @param {onRemoteRequestNext} next - Callback to advance the middleware chain
   * @param {MiddlewareContext} ctx - Local context for ths invocation of the middleware
   */
  onRemoteRequest(
    req: RemoteRequest,
    next: onRemoteRequestNext,
    ctx: MiddlewareContext
  ): Promise<RemoteRes>
}

/**
 * next callback to middleware methods that instrument http requests
 */
export type onRemoteRequestNext = (
  req: RemoteRequest,
  ctx: MiddlewareContext,
) => Promise<RemoteRes>

/**
 * Local context for invocation of middleware. Can be used to pass values from caller
 * to middleware callback invocations, or from a middleware to another middleware
 * further in the chain
 */
export interface MiddlewareContext {
  /**
   * Get a value from the context
   * @param key - Key of value
   * @returns Value for the key, or null if there is no value for the key
   */
  get(key: string): any | null

  /**
   * Set a value in the current context
   * @param key - key of value to set
   * @param value - the value to set
   */
  set(key: string, value: any): MiddlewareContext

  /**
   * Remove a value from the current context
   * @param key - Key of the value to remove from context
   */
  delete(key: string): MiddlewareContext
}

export enum Method {
  GET,
  POST
}

/**
 * A LanceDB Remote HTTP Request
 */
export interface RemoteRequest {
  uri: string
  method: Method
  headers: Record<string, string>
  params?: Record<string, string | number>
}

/**
 * A LanceDB Remote HTTP Response
 */
export interface RemoteRes {
  status: number
  headers: Record<string, string>
  body: () => Promise<any>
}

// TODO - should this be replaced by a diferent implementation?
export class SimpleMiddlewareContext implements MiddlewareContext {
  private context: Record<string, any> = {}

  get (key: string): any | null {
    const val: any = this.context[key]
    if (val === undefined) {
      return null
    }
    return val
  }

  set (key: string, value: any): MiddlewareContext {
    this.context[key] = value
    return this
  }

  delete (key: string): MiddlewareContext {
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete this.context[key]
    return this
  }
}
